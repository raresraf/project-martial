//
//  main.cpp
//  231A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <iostream>
using namespace std;

int main() {
  int n, total;
  int aux1, aux2, aux3, total2 = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> aux1 >> aux2 >> aux3;
    total = aux1 + aux2 + aux3;
    if (total >= 2) {
      total2++;
    }
  }

  cout << total2;
  return 0;
}
