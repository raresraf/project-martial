//
//  main.cpp
//  158A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

#include <iostream>
using namespace std;

int main() {
  int n, pos, aux, comp, ans = 0;
  cin >> n >> pos;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> aux;
    arr[i] = aux;
  }
  comp = arr[pos - 1];
  for (int i = 0; i < n; i++) {
    if (arr[i] >= comp && arr[i] > 0) {
      ans++;
    }
  }

  cout << ans;

  return 0;
}
