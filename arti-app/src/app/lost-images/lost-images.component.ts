import { Component } from '@angular/core';
import { ImageService } from '../services/image.service';
import { SharedService } from '../services/shared.service';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';

@Component({
  selector: 'app-lost-images',
  templateUrl: './lost-images.component.html',
  styleUrl: './lost-images.component.css'
})

export class LostImagesComponent {
  selectedFile: File | null = null;
  additionalParam: string = '';
  responseMessage: string = '';
  artistImages: any[] = [];
  closeImages: any[] = [];
  artistName: string = '';
  similarity: string = '';
  closestImageId: string = '';
  paintingName: string = '';
  fileName: string = ''; // for debugging, not shown in the template
  closestImageData: string = '';
  cols: number = 1;
  closenessThreshold: string = '';
  averageDistance: string = '';



  constructor(private imageService: ImageService, private sharedService: SharedService, private breakpointObserver: BreakpointObserver) { }

  ngOnInit() {
    this.sharedService.artistNameLP$.subscribe(name => this.artistName = name);
    this.sharedService.similarityLP$.subscribe(similarity => this.similarity = similarity);
    this.sharedService.closestImageIdLP$.subscribe(id => this.closestImageId = id);
    this.sharedService.paintingNameLP$.subscribe(name => this.paintingName = name);
    this.sharedService.fileNameLP$.subscribe(name => this.fileName = name); // for debugging
    this.sharedService.closestImageDataLP$.subscribe(images => this.closestImageData = images);
    this.sharedService.closeImagesLP$.subscribe(images => this.closeImages = images);
    this.sharedService.closenessThresholdLP$.subscribe(threshold => this.closenessThreshold = threshold);

    this.breakpointObserver.observe([
      Breakpoints.XSmall,
      Breakpoints.Small,
      Breakpoints.Medium,
      Breakpoints.Large,
      Breakpoints.XLarge,
    ]).subscribe(result => {
      if (result.breakpoints[Breakpoints.XSmall]) {
        this.cols = 1;
      } else if (result.breakpoints[Breakpoints.Small]) {
        this.cols = 2;
      } else if (result.breakpoints[Breakpoints.Medium]) {
        this.cols = 3;
      } else if (result.breakpoints[Breakpoints.Large]) {
        this.cols = 4;
      } else {
        this.cols = 5; // for extra large screens
      }
    });
  }

  getImageUrl(id: string): string {
    return this.imageService.getImageUrl(id);
  }
}
