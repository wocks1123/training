
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from exp.pascalvoc0712 import train_pascalvoc0712, eval_pascalvoc0712


def main():
    train_pascalvoc0712()
    eval_pascalvoc0712()


if __name__ == "__main__":
    main()

