//
//  main.cpp
//  263A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int aux, res;
  int mat[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> aux;
      // mat[i][j] = aux;
      if (aux == 1) {
        res = abs(2 - i) + abs(2 - j);
        cout << res;
      }
    }
  }
  return 0;
}
