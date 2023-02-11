import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { max } from 'rxjs';

export interface Tile {
  color: string;
  cols: number;
  rows: number;
  text?: string[] | undefined;
}

@Component({
  selector: 'app-diff',
  templateUrl: './diff.component.html',
  styleUrls: ['./diff.component.css']
})
export class DiffComponent {
  cpp1 = `#include <iostream>

using namespace std;

int main() {
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  return 0;
}
`

  cpp2 = `#include <iostream>

using namespace std;

int main() {
  // With a comment.
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  return 0;
}
`

  file_upload_tiles: Tile[] = [
    { cols: 1, rows: 1, color: '#ffe6e6' },
    { cols: 1, rows: 1, color: '#dcf0d5' },
  ];

  tiles: Tile[] = [
    { text: this.cpp1.split(/\r?\n/), cols: 1, rows: 10, color: "#FFFEFE" },
    { text: this.cpp2.split(/\r?\n/), cols: 1, rows: 10, color: "#FEFFFE" },
  ];


  fileName: string[] = ["", ""]
  constructor(private http: HttpClient) { }

  onFileSelected(event, id: number) {
    const file: File = event.target.files[0];
    if (file) {
      this.fileName[id] = file.name;
      let reader = new FileReader();
      reader.onload = (evt) => {
        this.tiles[id].text = evt.target?.result?.toString().split(/\r?\n/)
      }
      reader.readAsBinaryString(file);

      // TODO(raresraf): Upload them to backend!

      // const formData = new FormData();  
      // formData.append("thumbnail", file);  
      // const upload$ = this.http.post("/api/thumbnail-upload", formData);  
      // upload$.subscribe();  
    }
  }

  onFileSelected0(event) {  
    this.onFileSelected(event, 0)
  }

  onFileSelected1(event) {  
    this.onFileSelected(event, 1)
  }

  render_index = 0
  MaxRowsToRender(): number {
    let l1 = 0
    let l2 = 0
    if (this.tiles[0].text != undefined) {
      l1 = this.tiles[0].text.length
    }
    if (this.tiles[1].text != undefined) {
      l2 = this.tiles[1].text.length
    }
    return Math.max(l1, l2)
  }

  createRange(number){
    // return new Array(number);
    return new Array(number).fill(0)
      .map((n, index) => index + 1);
  }
}
