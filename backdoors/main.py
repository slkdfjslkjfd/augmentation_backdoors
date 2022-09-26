from backdoors.standard.standard_backdoor import run_standard_backdoor
from backdoors.gan_aug.gan_backdoor import run_gan_backdoor

def main(configs):

    attacks = {
        "standard": run_standard_backdoor,
        "gan": run_gan_backdoor
    }

    for attack, config in configs["attacks"].items():
        attacks[attack](config)

if __name__ == "__main__":

    from sys import argv
    import yaml

    CONFIG_FILE = "configs/" + argv[1]

    with open(CONFIG_FILE, "r") as f:
        c = f.read()
        configs = yaml.safe_load(c)
        print(c)
    
    main(configs)