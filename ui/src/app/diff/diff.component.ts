import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface Tile {
  color: string;
  cols: number;
  rows: number;
  text?: string | undefined;
}

@Component({
  selector: 'app-diff',
  templateUrl: './diff.component.html',
  styleUrls: ['./diff.component.css']
})
export class DiffComponent {
  cpp1 = `
  #include <iostream>

  using namespace std;

  int main() {
      cout << "Hello World!";
      return 0;
  }
  `

  cpp2 = `
  #include <iostream>

  using namespace std;

  int main() {
      // With a comment.
      cout << "Hello World!";
      return 0;
  }
  `

  file_upload_tiles: Tile[] = [
    { cols: 1, rows: 1, color: '#ffe6e6' },
    { cols: 1, rows: 1, color: '#dcf0d5' },
  ];

  tiles: Tile[] = [
    { text: this.cpp1, cols: 1, rows: 10, color: "#FFFEFE" },
    { text: this.cpp2, cols: 1, rows: 10, color: "#FEFFFE" },
  ];


  fileName: string[] = ["", ""]
  constructor(private http: HttpClient) { }

  onFileSelected(event, id: number) {
    const file: File = event.target.files[0];
    if (file) {
      this.fileName[id] = file.name;
      console.log("%s", this.fileName[id])
      let reader = new FileReader();
      reader.onload = (evt) => {
        this.tiles[id].text = evt.target?.result?.toString()
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
}
