import torch
import torch.nn as nn

class SELDLoss_ADPIT(object):
    """
    ADPIT loss = DOA(MSE) + SED(BCE) with permutation-invariant min selection (13 perms).
    - I/O shapes are identical to the original baseline.
    - SED logits are aligned to DCASE evaluation: act_scale * (||pred_xyz|| - sed_threshold).
    """
    def __init__(self,
                 doa_weight: float = 10.0,
                 sed_weight: float = 1.0,
                 act_scale: float = 8.0,
                 sed_threshold: float = 0.5):  # <-- baseline 평가에서 쓰는 magnitude threshold로 맞추세요
        super().__init__()
        self._mse = nn.MSELoss(reduction='none')
        self._bce = nn.BCEWithLogitsLoss(reduction='none')
        self.doa_weight = doa_weight
        self.sed_weight = sed_weight
        self.act_scale = act_scale
        self.sed_threshold = sed_threshold

    def _doa_mse_classwise(self, output_9C, target_9C):
        # output_9C, target_9C: [B, T, 9, C] -> returns [B, T, C]
        return self._mse(output_9C, target_9C).mean(dim=2)

    def _sed_bce_classwise(self, sed_logits_3C, sed_target_3C):
        # sed_logits_3C, sed_target_3C: [B, T, 3, C] -> returns [B, T, C]
        return self._bce(sed_logits_3C, sed_target_3C).mean(dim=2)

    def __call__(self, output, target):
        """
        Args:
            output: [B, T, 3*3*C]
            target: [B, T, 6, num_axis(>=4), C]  # 0: SED, 1:4: xyz, (opt) 4: distance
        Return:
            loss: scalar
        """
        B, T, _ = output.shape
        C = target.shape[-1]
        assert target.shape[2] == 6, "ADPIT requires 6 tracks (A0,B0,B1,C0,C1,C2)."
        assert target.shape[3] >= 4, "target num_axis must be >=4 (0:SED, 1:4=xyz)."

        def _masked_xyz(track_idx):
            sed = target[:, :, track_idx, 0:1, :]   # [B,T,1,C]
            xyz = target[:, :, track_idx, 1:4, :]   # [B,T,3,C]  (distance 무시)
            return sed * xyz                        # [B,T,3,C]

        # Track-wise masked xyz (A/B/C)
        A0 = _masked_xyz(0)
        B0 = _masked_xyz(1); B1 = _masked_xyz(2)
        C0 = _masked_xyz(3); C1 = _masked_xyz(4); C2 = _masked_xyz(5)

        # Track-wise SED targets (0/1)
        sed_A0 = target[:, :, 0, 0:1, :]
        sed_B0 = target[:, :, 1, 0:1, :]
        sed_B1 = target[:, :, 2, 0:1, :]
        sed_C0 = target[:, :, 3, 0:1, :]
        sed_C1 = target[:, :, 4, 0:1, :]
        sed_C2 = target[:, :, 5, 0:1, :]

        # DOA permutations -> [B,T,9,C]
        A0A0A0 = torch.cat((A0, A0, A0), dim=2)

        B0B0B1 = torch.cat((B0, B0, B1), dim=2)
        B0B1B0 = torch.cat((B0, B1, B0), dim=2)
        B0B1B1 = torch.cat((B0, B1, B1), dim=2)
        B1B0B0 = torch.cat((B1, B0, B0), dim=2)
        B1B0B1 = torch.cat((B1, B0, B1), dim=2)
        B1B1B0 = torch.cat((B1, B1, B0), dim=2)

        C0C1C2 = torch.cat((C0, C1, C2), dim=2)
        C0C2C1 = torch.cat((C0, C2, C1), dim=2)
        C1C0C2 = torch.cat((C1, C0, C2), dim=2)
        C1C2C0 = torch.cat((C1, C2, C0), dim=2)
        C2C0C1 = torch.cat((C2, C0, C1), dim=2)
        C2C1C0 = torch.cat((C2, C1, C0), dim=2)

        # SED permutations -> [B,T,3,C]
        sed_A0A0A0 = torch.cat((sed_A0, sed_A0, sed_A0), dim=2)

        sed_B0B0B1 = torch.cat((sed_B0, sed_B0, sed_B1), dim=2)
        sed_B0B1B0 = torch.cat((sed_B0, sed_B1, sed_B0), dim=2)
        sed_B0B1B1 = torch.cat((sed_B0, sed_B1, sed_B1), dim=2)
        sed_B1B0B0 = torch.cat((sed_B1, sed_B0, sed_B0), dim=2)
        sed_B1B0B1 = torch.cat((sed_B1, sed_B0, sed_B1), dim=2)
        sed_B1B1B0 = torch.cat((sed_B1, sed_B1, sed_B0), dim=2)

        sed_C0C1C2 = torch.cat((sed_C0, sed_C1, sed_C2), dim=2)
        sed_C0C2C1 = torch.cat((sed_C0, sed_C2, sed_C1), dim=2)
        sed_C1C0C2 = torch.cat((sed_C1, sed_C0, sed_C2), dim=2)
        sed_C1C2C0 = torch.cat((sed_C1, sed_C2, sed_C0), dim=2)
        sed_C2C0C1 = torch.cat((sed_C2, sed_C0, sed_C1), dim=2)
        sed_C2C1C0 = torch.cat((sed_C2, sed_C1, sed_C0), dim=2)

        # Output reshape: [B,T,9,C]
        out_9C = output.view(B, T, 9, C)

        # DOA pads (원본과 동일)
        pad4A = B0B0B1 + C0C1C2
        pad4B = A0A0A0 + C0C1C2
        pad4C = A0A0A0 + B0B0B1

        # DOA losses per permutation: [B,T,C]
        doa_0  = self._doa_mse_classwise(out_9C, A0A0A0 + pad4A)
        doa_1  = self._doa_mse_classwise(out_9C, B0B0B1 + pad4B)
        doa_2  = self._doa_mse_classwise(out_9C, B0B1B0 + pad4B)
        doa_3  = self._doa_mse_classwise(out_9C, B0B1B1 + pad4B)
        doa_4  = self._doa_mse_classwise(out_9C, B1B0B0 + pad4B)
        doa_5  = self._doa_mse_classwise(out_9C, B1B0B1 + pad4B)
        doa_6  = self._doa_mse_classwise(out_9C, B1B1B0 + pad4B)
        doa_7  = self._doa_mse_classwise(out_9C, C0C1C2 + pad4C)
        doa_8  = self._doa_mse_classwise(out_9C, C0C2C1 + pad4C)
        doa_9  = self._doa_mse_classwise(out_9C, C1C0C2 + pad4C)
        doa_10 = self._doa_mse_classwise(out_9C, C1C2C0 + pad4C)
        doa_11 = self._doa_mse_classwise(out_9C, C2C0C1 + pad4C)
        doa_12 = self._doa_mse_classwise(out_9C, C2C1C0 + pad4C)

        # SED logits from output: align to baseline threshold
        out_5d = out_9C.view(B, T, 3, 3, C)                  # [B,T,tracks=3,xyz=3,C]
        mag = out_5d.norm(dim=3)                             # [B,T,3,C]  (||pred_xyz||)
        sed_logits = self.act_scale * (mag - self.sed_threshold)  # allow negative/positive logits

        # SED BCE per permutation: [B,T,C]
        sed_0  = self._sed_bce_classwise(sed_logits, sed_A0A0A0)
        sed_1  = self._sed_bce_classwise(sed_logits, sed_B0B0B1)
        sed_2  = self._sed_bce_classwise(sed_logits, sed_B0B1B0)
        sed_3  = self._sed_bce_classwise(sed_logits, sed_B0B1B1)
        sed_4  = self._sed_bce_classwise(sed_logits, sed_B1B0B0)
        sed_5  = self._sed_bce_classwise(sed_logits, sed_B1B0B1)
        sed_6  = self._sed_bce_classwise(sed_logits, sed_B1B1B0)
        sed_7  = self._sed_bce_classwise(sed_logits, sed_C0C1C2)
        sed_8  = self._sed_bce_classwise(sed_logits, sed_C0C2C1)
        sed_9  = self._sed_bce_classwise(sed_logits, sed_C1C0C2)
        sed_10 = self._sed_bce_classwise(sed_logits, sed_C1C2C0)
        sed_11 = self._sed_bce_classwise(sed_logits, sed_C2C0C1)
        sed_12 = self._sed_bce_classwise(sed_logits, sed_C2C1C0)

        # Combine & PI-min
        totals = torch.stack((
            self.doa_weight*doa_0  + self.sed_weight*sed_0,
            self.doa_weight*doa_1  + self.sed_weight*sed_1,
            self.doa_weight*doa_2  + self.sed_weight*sed_2,
            self.doa_weight*doa_3  + self.sed_weight*sed_3,
            self.doa_weight*doa_4  + self.sed_weight*sed_4,
            self.doa_weight*doa_5  + self.sed_weight*sed_5,
            self.doa_weight*doa_6  + self.sed_weight*sed_6,
            self.doa_weight*doa_7  + self.sed_weight*sed_7,
            self.doa_weight*doa_8  + self.sed_weight*sed_8,
            self.doa_weight*doa_9  + self.sed_weight*sed_9,
            self.doa_weight*doa_10 + self.sed_weight*sed_10,
            self.doa_weight*doa_11 + self.sed_weight*sed_11,
            self.doa_weight*doa_12 + self.sed_weight*sed_12
        ), dim=0)  # [13,B,T,C]

        min_idx = totals.min(dim=0).indices  # [B,T,C]
        loss = sum(totals[i] * (min_idx == i) for i in range(13)).mean()
        return loss
