import { Component } from '@angular/core';
import { ImageService } from '../services/image.service';
import { SharedService } from '../services/shared.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-image-upload',
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.css']
})
export class ImageUploadComponent {
  imagePreview: string | ArrayBuffer | null = null;
  selectedFile: File | null = null;
  artistName : string = '';
  similarImages: any[] = [];
  similarity :string = '';
  closest_image_id = '';
  painting_name = '';
  file_name = '';  //dont show this just for debugging


  constructor(private imageService: ImageService, private sharedService: SharedService, private router:Router) { }

  onFileSelected(event: any) {
    const fileInput = event.target as HTMLInputElement;
    if (fileInput.files && fileInput.files.length > 0) {
      this.selectedFile = fileInput.files[0];

      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result;
      };
      reader.readAsDataURL(this.selectedFile);

      const currentUrl = this.router.url;
      if (currentUrl === '/info') {
        this.getImageInfos();
      } else if (currentUrl === '/lostImages') {
        this.getLostImageInfos();
      } else {
        console.error('Unexpected URL:', currentUrl);
      }
    }
  }

  getImageInfos() {
    if (this.selectedFile) {
      this.imageService.inferArtistName(this.selectedFile)
        .subscribe(responseCNN => {
          this.sharedService.setArtistName(responseCNN.artist_name);
          this.sharedService.setArtistImages(responseCNN.random_paintings);

          
          if (this.selectedFile) {
            this.imageService.inferSimilarity(this.selectedFile, responseCNN.artist_name)
            .subscribe(responseSiamese => {
                this.sharedService.setSimilarity(responseSiamese.similarity);
                this.sharedService.setClosestImageId(responseSiamese.closest_image_id);
                this.sharedService.setPaintingName(responseSiamese.painting_name);
                this.sharedService.setFileName(responseSiamese.file_name);
                this.sharedService.setClosestImageData(responseSiamese.closest_image_data);
                this.sharedService.setCloseImages(responseSiamese.close_paintings);
                this.sharedService.setClosenessThresold(responseSiamese.closeness_threshold);
                this.sharedService.setAverageDistance(responseSiamese.average_distance)
            },
            error => {
              console.error('Error inferring similarity:', error);
              alert('Error inferring similarity: ' + error.message);
            });

          }else{
            console.error('No file selected');
          }
        },
        error => {
          console.error('Error inferring artist name:', error);
          alert('Error inferring artist name: ' + error.message);
        });
    }
    else{
      console.error('No file selected');
    }
  }

  getLostImageInfos(){
    if (this.selectedFile) {
      this.imageService.inferLostImage(this.selectedFile)
        .subscribe(responseSiamese => {
          this.sharedService.setArtistNameLP(responseSiamese.artist_name);
          this.sharedService.setSimilarityLP(responseSiamese.similarity);
          this.sharedService.setClosestImageIdLP(responseSiamese.closest_image_id);
          this.sharedService.setPaintingNameLP(responseSiamese.painting_name);
          this.sharedService.setFileNameLP(responseSiamese.file_name);
          this.sharedService.setClosestImageDataLP(responseSiamese.closest_image_data);
          this.sharedService.setCloseImagesLP(responseSiamese.close_paintings);
          this.sharedService.setClosenessThresholdLP(responseSiamese.closeness_threshold);
          console.dir(responseSiamese);
        },
        error => {
          console.error('Error inferring lost image:', error);
          alert('Error inferring lost image: ' + error.message);
        });
    }
    else{
      console.error('No file selected');
    }
    
  }

}


