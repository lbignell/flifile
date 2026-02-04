"""
Python support for the Lambert Instruments .fli file

(c) R.Harkes NKI
Enhanced with memory-mapped frame access and batch processing by L. Bignell, The Australian National University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import os
import zlib
from pathlib import Path
from typing import Any, Union, Iterator, Optional
import numpy as np
import numpy.typing as npt

from .datatypes import Datatypes, Packing, np_dtypes
from .readheader import readheader, telldatainfo


class FliFile:
    """
    Lambert Instruments .fli file with memory-efficient frame access
    
    Contains:
    - header: dictionary with all header entries
    - path: pathlib.Path to the file
    - datainfo: DataInfo object with file metadata
    
    New features:
    - Lazy loading: Data not loaded until accessed
    - Memory-mapped file access for large files
    - Array-like indexing: file[0], file[10:20], file[:100]
    - Batch processing with iter_batches()
    
    Hidden:
    - _bg: to store the background
    - _datastart: pointer to the start of the data
    - _data: cached full data (only if getdata() called)
    - _mmap: memory-mapped file for lazy frame access
    """

    def __init__(
        self, 
        filepath: Union[str, os.PathLike[Any]], 
        lazy: bool = True
    ) -> None:
        """
        Open a .fli file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the .fli file
        lazy : bool
            If True, don't load data into memory until needed (default: True)
            If False, use original behavior (load everything)
        """
        # Open file
        if isinstance(filepath, str):
            self.path = Path(filepath)
        elif isinstance(filepath, Path):
            self.path = filepath
        else:
            raise ValueError("not a valid filename")
        if self.path.suffix != ".fli":
            raise ValueError("Not a valid extension")
            
        self.log = logging.getLogger("flifile")
        self.header, self._datastart = readheader(self.path)
        self.datainfo = telldatainfo(self.header)
        
        # Background storage
        self._bg: npt.NDArray[np_dtypes] = np.array(
            [], dtype=self.datainfo.BGType.nptype
        )
        
        # Lazy loading setup
        self._lazy = lazy
        self._data: Optional[npt.NDArray[np_dtypes]] = None
        self._mmap: Optional[np.memmap] = None
        
        # Calculate frame size for memory-mapped access
        # IMSize is (ch, x, y, z, ph, t, freq)
        self._pixels_per_2d_frame = self.datainfo.IMSize[1] * self.datainfo.IMSize[2]
        
        if self.datainfo.IMType.bits == 12:
            self._bytes_per_2d_frame = int(self._pixels_per_2d_frame * 1.5)
        else:
            self._bytes_per_2d_frame = self._pixels_per_2d_frame * (self.datainfo.IMType.bits // 8)
        
        # Setup memory map if lazy loading and not compressed
        if self._lazy and self.datainfo.Compression == 0:
            self._setup_memmap()

    def _setup_memmap(self) -> None:
        """Setup memory-mapped file access."""
        try:
            self._mmap = np.memmap(self.path, dtype=np.uint8, mode='r')
        except Exception as e:
            self.log.warning(f"Could not create memory map: {e}. Falling back to eager loading.")
            self._lazy = False
            self._mmap = None

    def __len__(self) -> int:
        """Return number of time frames."""
        return self.datainfo.IMSize[5]  # t dimension

    def __getitem__(
        self, 
        key: Union[int, slice]
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """
        Array-like indexing for time frames.
        
        Examples:
        ---------
        >>> file[0]              # First frame
        >>> file[10:20]          # Frames 10-19
        >>> file[-1]             # Last frame
        >>> file[::10]           # Every 10th frame
        
        Returns:
        --------
        For single index: 2D array (y, x) or squeezed multi-dimensional
        For slice: 3D array (frames, y, x) or squeezed multi-dimensional
        """
        if isinstance(key, int):
            # Single frame
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Frame index {key} out of range [0, {len(self)})")
            return self.getframe(timestamp=key, squeeze=True)
        
        elif isinstance(key, slice):
            # Slice of frames
            start, stop, step = key.indices(len(self))
            
            if step == 1:
                # Contiguous slice - can be efficient
                return self._get_frame_range(start, stop)
            else:
                # Non-contiguous - get individual frames
                frames = [self.getframe(timestamp=i, squeeze=True) for i in range(start, stop, step)]
                return np.stack(frames, axis=0)
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key).__name__}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close memory-mapped file if open."""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

    def iter_batches(
        self, 
        batch_size: int = 50,
        subtractbackground: bool = False,
        squeeze: bool = True
    ) -> Iterator[np.ndarray[Any, np.dtype[np_dtypes]]]:
        """
        Iterate through data in batches (memory-efficient).
        
        Parameters:
        -----------
        batch_size : int
            Number of frames per batch (default: 50)
        subtractbackground : bool
            Subtract background from each batch
        squeeze : bool
            Remove singleton dimensions
            
        Yields:
        -------
        numpy.ndarray
            Batch of frames
            
        Example:
        --------
        >>> with FliFile('data.fli') as f:
        ...     for batch in f.iter_batches(batch_size=100):
        ...         process(batch)
        """
        num_frames = len(self)
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            yield self._get_frame_range(i, end, subtractbackground, squeeze)

    def getdata(
        self, subtractbackground: bool = True, squeeze: bool = True
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """
        Returns the data from the .fli file. If squeeze is False the data is returned with these dimensions:
        frequency,time,phase,z,y,x,channel
        
        :param subtractbackground: Subtract the background from the image data
        :param squeeze: Return data without singleton dimensions in x,y,ph,t,z,fr,c order
        :return: numpy.ndarray
        """
        # Return cached data if available
        if self._data is not None:
            data = self._data.copy()
            if subtractbackground and self.datainfo.BG_present:
                self._bg = self.getbackground(squeeze=False)
                mask = np.where(data < self._bg)
                data = data - self._bg
                data[mask] = 0
            if squeeze:
                data = np.squeeze(data.transpose((5, 4, 2, 1, 3, 0, 6)))
            return data
        
        # Original implementation
        if not self.datainfo.BG_present:
            subtractbackground = False
        datasize = int(np.prod(self.datainfo.IMSize, dtype=np.uint64))
        
        if self.datainfo.Compression > 0:
            fid = self.path.open(mode="rb")
            fid.seek(self._datastart)
            dcmp = zlib.decompressobj(32 + zlib.MAX_WBITS)
            data: npt.NDArray[np_dtypes] = np.frombuffer(
                dcmp.decompress(fid.read()), dtype=self.datainfo.IMType.nptype
            )
            bg = data[datasize:]
            self._bg = bg.reshape(self.datainfo.BGSize[::-1])
            data = data[:datasize]
        else:
            data = self._get_data_from_file(
                offset=self._datastart,
                datatype=self.datainfo.IMType,
                datasize=datasize,
            )
            
        if self.datainfo.IMType.bits == 12:
            data = self._convert_12_bit(data, datatype=self.datainfo.IMType)
            
        data = data.reshape(self.datainfo.IMSize[::-1])
        
        # Cache the data
        self._data = data.copy()
        
        if subtractbackground:
            self._bg = self.getbackground(squeeze=False)
            mask = np.where(data < self._bg)
            data = data - self._bg
            data[mask] = 0
            
        if squeeze:
            data = np.squeeze(data.transpose((5, 4, 2, 1, 3, 0, 6)))

        return data

    def getbackground(self, squeeze: bool = True) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """
        Returns the background data from the .fli file. If squeeze is False the data is returned with these dimensions:
        frequency,time,phase,z,y,x,channel
        :param squeeze: Return data without singleton dimensions in x,y,ph,t,z,fr,c order
        :return: numpy.ndarray
        """
        if not self.datainfo.BG_present:
            self.log.warning("WARNING: No background present in file")
            return np.array([])
        if self._bg.size != 0:
            data = self._bg
        else:
            if self.datainfo.Compression > 0:
                self.log.warning(
                    "WARNING: Getting background before getting data is inefficient in compressed files."
                )
                self.getdata(subtractbackground=True, squeeze=False)
                data = self._bg
            else:
                offset = (
                    self._datastart
                    + (self.datainfo.IMType.bits * int(np.prod(self.datainfo.IMSize, dtype=np.uint64))) / 8
                )
                datasize = int(np.prod(self.datainfo.BGSize, dtype=np.uint64))
                data = self._get_data_from_file(
                    offset=int(offset), datatype=self.datainfo.BGType, datasize=datasize
                )
                if self.datainfo.BGType.bits == 12:# 12 bit per pixel packed per 2 in 3 bytes
                    data = self._convert_12_bit(data, datatype=self.datainfo.BGType)
                data = data.reshape(self.datainfo.BGSize[::-1])
        if squeeze:
            return np.squeeze(data.transpose((5, 4, 2, 1, 3, 0, 6)))
        else:
            return data

    def getframe(
        self,
        channel: int = 0,
        z: int = 0,
        phase: int = 0,
        timestamp: int = 0,
        frequency: int = 0,
        subtractbackground: bool = False,
        squeeze: bool = True,
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """
        Get a single frame from the .fli file using memory-mapped access.
        
        Parameters:
        -----------
        channel : int
            Channel index (default: 0)
        z : int
            Z-plane index (default: 0)
        phase : int
            Phase index (default: 0)
        timestamp : int
            Time frame index (default: 0)
        frequency : int
            Frequency index (default: 0)
        subtractbackground : bool
            Subtract background (default: False)
        squeeze : bool
            Remove singleton dimensions (default: True)
            
        Returns:
        --------
        numpy.ndarray
            Single frame, typically 2D (y, x) if squeezed
        """
        # Validate indices
        if channel > (self.datainfo.IMSize[0] - 1) or channel < 0:
            raise IndexError(f"Channel {channel} out of range [0, {self.datainfo.IMSize[0]})")
        if z > (self.datainfo.IMSize[3] - 1) or z < 0:
            raise IndexError(f"Z {z} out of range [0, {self.datainfo.IMSize[3]})")
        if phase > (self.datainfo.IMSize[4] - 1) or phase < 0:
            raise IndexError(f"Phase {phase} out of range [0, {self.datainfo.IMSize[4]})")
        if timestamp > (self.datainfo.IMSize[5] - 1) or timestamp < 0:
            raise IndexError(f"Timestamp {timestamp} out of range [0, {self.datainfo.IMSize[5]})")
        if frequency > (self.datainfo.IMSize[6] - 1) or frequency < 0:
            raise IndexError(f"Frequency {frequency} out of range [0, {self.datainfo.IMSize[6]})")

        # For compressed files or if data already loaded, use getdata
        if self.datainfo.Compression > 0 or self._data is not None:
            data = self.getdata(subtractbackground=False, squeeze=False)
            frame = data[frequency, timestamp, phase, z, :, :, channel]
            
            if subtractbackground and self.datainfo.BG_present:
                bg = self.getbackground(squeeze=False)
                bg_frame = bg[frequency, 0, phase, z, :, :, channel] if bg.size > 0 else 0
                frame = np.clip(frame - bg_frame, 0, None)
                
            if squeeze:
                return np.squeeze(frame)
            return frame

        # Use memory-mapped access for uncompressed files
        if not self._lazy or self._mmap is None:
            # Fallback to loading everything
            return self.getframe(channel, z, phase, timestamp, frequency, subtractbackground, squeeze)

        # Calculate frame position in file
        # IMSize is (ch, x, y, z, ph, t, freq)
        frame_index = self._calculate_frame_index(channel, z, phase, timestamp, frequency)
        
        # Read frame from memory map
        start_byte = self._datastart + frame_index * self._bytes_per_2d_frame
        end_byte = start_byte + self._bytes_per_2d_frame
        
        frame_bytes = self._mmap[start_byte:end_byte]
        
        # Decode based on bit depth
        if self.datainfo.IMType.bits == 12:
            frame = self._decode_12bit_frame(frame_bytes, self.datainfo.IMType)
        else:
            frame = np.frombuffer(frame_bytes, dtype=self.datainfo.IMType.nptype)
        
        # Reshape to 2D
        frame = frame.reshape(self.datainfo.IMSize[2], self.datainfo.IMSize[1])
        
        if subtractbackground and self.datainfo.BG_present:
            bg = self.getbackground(squeeze=False)
            bg_frame = bg[frequency, 0, phase, z, :, :, channel] if bg.size > 0 else 0
            frame = np.clip(frame.astype(np.int32) - bg_frame.astype(np.int32), 0, None).astype(frame.dtype)
        
        if squeeze:
            return np.squeeze(frame)
        return frame

    def _get_frame_range(
        self,
        start: int,
        end: int,
        subtractbackground: bool = False,
        squeeze: bool = True
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """
        Get a range of frames efficiently.
        
        Parameters:
        -----------
        start : int
            Start frame index (inclusive)
        end : int
            End frame index (exclusive)
        subtractbackground : bool
            Subtract background
        squeeze : bool
            Remove singleton dimensions
            
        Returns:
        --------
        numpy.ndarray
            Array of frames (num_frames, y, x) if squeezed
        """
        num_frames = end - start
        
        # For compressed or cached data, use getdata
        if self.datainfo.Compression > 0 or self._data is not None:
            data = self.getdata(subtractbackground=False, squeeze=False)
            frames = data[0, start:end, 0, 0, :, :, 0]
            
            if subtractbackground and self.datainfo.BG_present:
                bg = self.getbackground(squeeze=False)
                bg_frame = bg[0, 0, 0, 0, :, :, 0] if bg.size > 0 else 0
                frames = np.clip(frames - bg_frame, 0, None)
                
            if squeeze:
                return np.squeeze(frames)
            return frames

        # Memory-mapped batch read
        if self._lazy and self._mmap is not None:
            # Calculate byte range
            frame_start_idx = self._calculate_frame_index(0, 0, 0, start, 0)
            start_byte = self._datastart + frame_start_idx * self._bytes_per_2d_frame
            end_byte = start_byte + num_frames * self._bytes_per_2d_frame
            
            batch_bytes = self._mmap[start_byte:end_byte]
            
            # Decode
            if self.datainfo.IMType.bits == 12:
                pixels = self._decode_12bit_batch(batch_bytes, self.datainfo.IMType, num_frames)
            else:
                pixels = np.frombuffer(batch_bytes, dtype=self.datainfo.IMType.nptype)
            
            # Reshape
            frames = pixels.reshape(num_frames, self.datainfo.IMSize[2], self.datainfo.IMSize[1])
            
            if subtractbackground and self.datainfo.BG_present:
                bg = self.getbackground(squeeze=False)
                bg_frame = bg[0, 0, 0, 0, :, :, 0] if bg.size > 0 else 0
                frames = np.clip(frames.astype(np.int32) - bg_frame.astype(np.int32), 0, None).astype(frames.dtype)
            
            if squeeze:
                return np.squeeze(frames)
            return frames
        
        # Fallback: get individual frames
        frames = [self.getframe(timestamp=i, squeeze=True) for i in range(start, end)]
        return np.stack(frames, axis=0)

    def _calculate_frame_index(
        self, channel: int, z: int, phase: int, timestamp: int, frequency: int
    ) -> int:
        """
        Calculate linear frame index from multi-dimensional indices.
        
        Data layout: (freq, time, phase, z, y, x, channel)
        """
        # IMSize is (ch, x, y, z, ph, t, freq)
        ch_size, x_size, y_size, z_size, ph_size, t_size, freq_size = self.datainfo.IMSize
        
        # Calculate linear index
        idx = (
            frequency * (t_size * ph_size * z_size * ch_size) +
            timestamp * (ph_size * z_size * ch_size) +
            phase * (z_size * ch_size) +
            z * ch_size +
            channel
        )
        
        return idx

    def _decode_12bit_frame(
        self, 
        frame_bytes: np.ndarray, 
        datatype: Datatypes
    ) -> np.ndarray:
        """Decode a single 12-bit packed frame."""
        num_groups = len(frame_bytes) // 3
        groups = frame_bytes[:num_groups * 3].reshape(-1, 3)
        
        pixels_per_frame = self._pixels_per_2d_frame
        pixels = np.empty(pixels_per_frame, dtype=np.uint16)
        
        byte1 = groups[:, 0]
        byte2 = groups[:, 1]
        byte3 = groups[:, 2]
        
        if datatype.packing == Packing.LSB:
            # Even pixels: byte1 + ((byte2 & 0x0F) << 8)
            pixels[0::2] = byte1.astype(np.uint16) + (
                (byte2 & 0x0F).astype(np.uint16) << 8
            )
            # Odd pixels: (byte3 << 4) + (byte2 >> 4)
            pixels[1::2] = (byte3.astype(np.uint16) << 4) + (byte2 >> 4).astype(np.uint16)
        elif datatype.packing == Packing.MSB:
            # Even pixels: (byte1 << 4) + (byte2 >> 4)
            pixels[0::2] = (byte1.astype(np.uint16) << 4) + (byte2 >> 4).astype(np.uint16)
            # Odd pixels: ((byte2 & 0x0F) << 8) + byte3
            pixels[1::2] = ((byte2 & 0x0F).astype(np.uint16) << 8) + byte3.astype(np.uint16)
        else:
            raise ValueError("Data has no valid packing type")
        
        return pixels

    def _decode_12bit_batch(
        self,
        batch_bytes: np.ndarray,
        datatype: Datatypes,
        num_frames: int
    ) -> np.ndarray:
        """Decode multiple 12-bit packed frames at once."""
        num_groups = len(batch_bytes) // 3
        groups = batch_bytes[:num_groups * 3].reshape(-1, 3)
        
        total_pixels = num_frames * self._pixels_per_2d_frame
        pixels = np.empty(total_pixels, dtype=np.uint16)
        
        byte1 = groups[:, 0]
        byte2 = groups[:, 1]
        byte3 = groups[:, 2]
        
        if datatype.packing == Packing.LSB:
            pixels[0::2] = byte1.astype(np.uint16) + (
                (byte2 & 0x0F).astype(np.uint16) << 8
            )
            pixels[1::2] = (byte3.astype(np.uint16) << 4) + (byte2 >> 4).astype(np.uint16)
        elif datatype.packing == Packing.MSB:
            pixels[0::2] = (byte1.astype(np.uint16) << 4) + (byte2 >> 4).astype(np.uint16)
            pixels[1::2] = ((byte2 & 0x0F).astype(np.uint16) << 8) + byte3.astype(np.uint16)
        else:
            raise ValueError("Data has no valid packing type")
        
        return pixels

    @staticmethod
    def _convert_12_bit(
        data: np.ndarray[Any, np.dtype[np_dtypes]], datatype: Datatypes
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """Convert 12-bit packed data (used by getdata for full file load)."""
        datasize = int((data.size / 3) * 2)
        byte1 = data[0::3]
        byte2 = data[1::3]
        byte3 = data[2::3]
        result = np.zeros(datasize, dtype=datatype.nptype)
        
        if datatype.packing == Packing.LSB:
            result[0::2] = byte1.astype(np.uint16) + np.left_shift(
                np.left_shift(byte2, 4).astype(np.uint8).astype(np.uint16), 4
            )
            result[1::2] = np.left_shift(byte3.astype(np.uint16), 4) + np.right_shift(
                byte2, 4
            ).astype(np.uint8)
        elif datatype.packing == Packing.MSB:
            result[0::2] = np.left_shift(byte1.astype(np.uint16), 4) + np.right_shift(
                byte2, 4
            ).astype(np.uint8)
            result[1::2] = np.left_shift(
                np.left_shift(byte2, 4).astype(np.uint8).astype(np.uint16), 4
            ) + byte3.astype(np.uint16)
        else:
            raise ValueError("Data has no valid packing type")
        return result

    def _get_data_from_file(
        self, offset: int, datatype: Datatypes, datasize: int
    ) -> np.ndarray[Any, np.dtype[np_dtypes]]:
        """Read data from file at given offset."""
        if datatype.bits == 12:
            count = int(3 * (datasize / 2))
            result12: npt.NDArray[np_dtypes] = np.fromfile(
                self.path, offset=offset, dtype=np.uint8, count=count
            )
            return result12
        result: npt.NDArray[np_dtypes] = np.fromfile(
            self.path, offset=offset, dtype=datatype.nptype, count=datasize
        )
        return result

    def __str__(self) -> str:
        return self.path.name

    def __repr__(self) -> str:
        return f"FliFile('{self.path}', {len(self)} frames, {self.datainfo.IMSize[1]}x{self.datainfo.IMSize[2]})"
