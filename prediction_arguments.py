import argparse

def prediction_args():
    
    parser = argparse.ArgumentParser(description='predictions')
    ###positional
    parser.add_argument("path_to_image", help = "choose your path" )
    parser.add_argument("checkpoint", help = "choose your stored model")
    ###optional
    parser.add_argument("--top_k", default = 5, type = int, help = "top k classes")
    parser.add_argument("--gpu", default = False, action = "store_true", help = "gpu is False by default")
    parser.add_argument("--category_names", default="cat_to_name.json", type = str, help = "actual names of the categories")         
                        
                        
                      

    
    ###
    parser.parse_args()
    return parser
    
    
    

def main():
    print("this is prediction arguments")
    
if __name__== "__main__":
    main()


