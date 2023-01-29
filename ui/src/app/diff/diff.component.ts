import { Component } from '@angular/core';

export interface Tile {
  color: string;
  text: string;
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
   int a;
   // read a.
   cin >> a;
   if (a % 2 == 0)
     // Check if a is even.
     cout << "even";
   else
     // Is odd.
     cout << "odd";

   return 0;
 }
  `
  cpp2 = `
  #include <iostream>

  using namespace std;
  
  int main() {
    int a;
    // read a.
    cin >> a;
    if (a % 2 == 0)
      // Check even.
      cout << "even";
    else
      // Check odd.
      cout << "odd";
    return 0;
  }  
  `

  tiles: Tile[] = [
    {text: this.cpp1, color: '#ffe6e6'},
    {text: this.cpp2, color: '#dcf0d5'},
  ];
}
