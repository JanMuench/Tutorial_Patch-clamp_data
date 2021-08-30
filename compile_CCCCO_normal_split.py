import pickle
import pystan

def save(obj, filename, regress_code_kinetic):
    """Save compiled models for reuse."""


    with open(filename+".pic", 'wb') as pointer:
        pickle.dump(obj, pointer, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename+".txt", "w") as file:
        file.write(regress_code_kinetic)



def main():
    print(pystan.__version__)
    with open("KF"+".txt", "r") as file:
        KF_code = file.read()

    print("constructiong the stan code")
    filename = "KF_CCCO"
    #print(filename + ".txt")
    print("Compile now \n" + KF_code)
    extra_compile_args = ["-pthread", "-DSTAN_THREADS"]
    #print(extra_compile_args)
    model = pystan.StanModel(model_code=KF_code,
                             extra_compile_args=extra_compile_args)

    save(model, filename, KF_code)
    print("Finished")

if __name__ == "__main__":
    main()
