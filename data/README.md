Put data in this folder.

The directory structure will be arranged as:
```
data
   |- train
      |- dataset1
         |- video
            |- video1.mp4
            |- ...
         |- metadata.csv
      |- dataset2
         |- video
            |- video1.mp4
            |- ...
         |- metadata.csv
      |- ...
   |- eval
      |- DAVIS
         |- JPEGImages
            |- 480p
               |- <video_name>
                  |- 00000.jpg
                  |- 00001.jpg
         |- Annotations
            |- 480p
               |- <video_name>
                  |- 00000.png
                  |- 00001.png   
         |- ...
```