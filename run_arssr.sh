#!/bin/bash
# run_arssr.sh
# ArSSR 4x 연속 추론 스크립트 (flair, t1ce)

CSV_PATH="/dshome/ddualab/dongnyeok/arssr/test.csv"
LR_BASE="/data/BraTS20_Degraded_4x_5"
GT_BASE="/data/BraTS2020_TrainingData"
OUTPUT_BASE="/dshome/ddualab/dongnyeok/arssr/output"
MODEL_PATH="./pre_trained_models/ArSSR_RDN.pkl"
MODALITIES=("flair" "t1ce")

# CSV에서 BraTS20ID 컬럼 읽기 (헤더 제외, 공백/CR 제거)
PATIENT_IDS=$(tail -n +2 "$CSV_PATH" | awk -F',' '{print $1}' | tr -d ' \r')

for PATIENT_ID in $PATIENT_IDS; do
    INPUT_DIR="${LR_BASE}/${PATIENT_ID}"
    GT_DIR="${GT_BASE}/${PATIENT_ID}"
    OUTPUT_DIR="${OUTPUT_BASE}/${PATIENT_ID}"
    mkdir -p "$OUTPUT_DIR"

    for MODALITY in "${MODALITIES[@]}"; do
        INPUT_FILE="${INPUT_DIR}/${PATIENT_ID}_${MODALITY}.nii"

        # 입력 파일 존재 여부 확인
        if [ ! -f "$INPUT_FILE" ]; then
            echo "[SKIP] Not found: $INPUT_FILE"
            continue
        fi

        echo "[RUN] ${PATIENT_ID} / ${MODALITY}"

        # 모달리티 파일 1개씩 처리하기 위해 임시 폴더 사용
        TMP_INPUT=$(mktemp -d)
        ln -s "$INPUT_FILE" "${TMP_INPUT}/${PATIENT_ID}_${MODALITY}.nii"

        python test.py \
            -input_path "$TMP_INPUT" \
            -output_path "$OUTPUT_DIR" \
            -gt_path "$GT_DIR" \
            -encoder RDN \
            -pre_trained_model "$MODEL_PATH" \
            -scale 4.0 \
            -aniso_axis 0 \
            -target_size 155 \
            -is_gpu 1 \
            -gpu 0

        rm -rf "$TMP_INPUT"

        # 결과 파일명 정리: ArSSR_RDN_recon_4d0x_*.nii -> BraTS20_Training_*_flair.nii
        RECON_FILE="${OUTPUT_DIR}/ArSSR_RDN_recon_4d0x_${PATIENT_ID}_${MODALITY}.nii"
        FINAL_FILE="${OUTPUT_DIR}/${PATIENT_ID}_${MODALITY}.nii"
        if [ -f "$RECON_FILE" ]; then
            mv "$RECON_FILE" "$FINAL_FILE"
            echo "[DONE] Saved: $FINAL_FILE"
        else
            echo "[WARN] Expected output not found: $RECON_FILE"
        fi
    done
done

echo "모든 환자 처리 완료."
