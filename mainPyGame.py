import pygame
import os
 
_image_library = {}
def get_image(path):
        global _image_library
        image = _image_library.get(path)
        if image == None:
                canonicalized_path = path.replace('/', os.sep).replace('', os.sep)
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (900, 1600))
                _image_library[path] = image
        return image
 
pygame.init()
screen = pygame.display.set_mode(( 900 , 1600))
done = False
clock = pygame.time.Clock()

DeltaTime = 0
count = 1
pathBeg = 'Background/Untitled-Artwork-'
fileExt = '.png'
initial = True
while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True
        
        DeltaTime = DeltaTime + clock.get_time()
        filePathStr = pathBeg + str(count) + fileExt
        
        if (initial == True):
            screen.blit(get_image(filePathStr), (0, 0))
            count = count + 1
            initial = False
        if(DeltaTime>1000):
            screen.blit(get_image(filePathStr), (0, 0))
            count = count + 1
            DeltaTime = 0
            
        if (count > 41):
            count = 1
            DeltaTime = 0

        pygame.display.flip()
        clock.tick(60)