import os
from spdn.trainers.n2_trainer import train_all_n2


def main():
    patient_count = 40

    override_dict = {
        "training": {
            "ablation": f"patient_count/{patient_count}_patients",
            "n_patients": patient_count,
        }
    }

    N2_PATH = os.environ.get("N2_CONFIG_PATH")
    train_all_n2(config_path=N2_PATH, ssm=False, override_config=override_dict)
    train_all_n2(config_path=N2_PATH, ssm=True, override_config=override_dict)


if __name__ == "__main__":
    main()
