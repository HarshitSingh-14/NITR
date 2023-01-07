# Stack overflow 
import pygame


def init():
    pygame.init()
    window = pygame.display.set_mode((500, 500))
 
 


# tells if key is correct pressed 
def getKey(keyName):
    ans = False
    for event in pygame.event.get(): pass

    keyInput = pygame.key.get_pressed()


    # output Format
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    # print('K_{}'.format(keyName))
 
    if keyInput[myKey]:
        ans = True
    
    #
    pygame.display.update()
    return ans
 
 
def main():
    if getKey("LEFT") and getKey("LSHIFT"):
        print("Left and key pressed")
    if getKey("LEFT") :
        print("Left")
    if getKey("LSHIFT") :
        print("LeftSHIFT")
    if getKey("RIGHT"):
        print("Right key Pressed")
 
    




# by default main  
if __name__ == "__main__":
    init()
    while True:
        main()