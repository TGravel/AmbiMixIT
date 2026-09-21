import torch

from lib.utils import flatten_sources, unflatten_sources


class Transform:
    def __init__(self, stft_frame_size, stft_hop_size, device):
        self.stft_frame_size = stft_frame_size
        self.stft_hop_size = stft_hop_size

        self.hann_window = torch.hann_window(
            self.stft_frame_size,
            periodic=True,
            device=device
        )

        self.num_frequency_bins = self.stft_frame_size // 2 + 1

    def stft(self, wave):
        wave_flat = flatten_sources(wave) # Originalen Entwickler gehen von (B, S, T) statt (B, C, T) aus?
        complex_flat = torch.stft(
            wave_flat,
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            return_complex=True
        )
        complex = unflatten_sources(complex_flat, num_sources=wave.size(1)) 
        mag, phase = complex.abs(), complex.angle()
        return mag, phase

    def stft_multichannel(self, wave):
        B, C_in, L = wave.shape
        wave_reshaped = wave.reshape(B * C_in, L) # Sollte möglich sein weil torch.stft() alle Zeilen einzeln und unabhängig umwandelt
        complex_stft = torch.stft(
            wave_reshaped, 
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            return_complex=True,
            normalized=False
        ) # [B * C, FrequencyBins, TimeFrames]
        mag_batched = torch.abs(complex_stft)
        phase_batched = torch.angle(complex_stft)
        num_batched_signals, F, T_f = mag_batched.shape
        mag = mag_batched.view(B, C_in, F, T_f)
        phase = phase_batched.view(B, C_in, F, T_f)
        return mag, phase

    def istft_multichannel_multisource(self, mag, phase, length):

        B, S, C_out, F, T_f = mag.shape

        mag_reshaped = mag.reshape(B * S * C_out, F, T_f)
        phase_reshaped = phase.reshape(B * S * C_out, F, T_f)

        complex_stft_reshaped = torch.complex(
            real=mag_reshaped * torch.cos(phase_reshaped), # statt cos(phase_reshaped)
            imag=mag_reshaped * torch.sin(phase_reshaped)  # statt sin(phase_reshaped)
        )

        wave_reshaped = torch.istft(
            complex_stft_reshaped,
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            length=length, # transform back to original wavelength segments
            normalized=False # to be consistent with STFT
        ) # [B * S * C_out, L]
        
        wave_final = wave_reshaped.view(B, S, C_out, length)
        return wave_final

    def istft(self, mag, phase, length):
        complex = torch.complex(
            real=mag * phase.cos(),
            imag=mag * phase.sin()
        )
        complex_flat = flatten_sources(complex)
        wave_flat = torch.istft(
            complex_flat,
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            length=length
        )
        return unflatten_sources(wave_flat, num_sources=mag.size(1))
