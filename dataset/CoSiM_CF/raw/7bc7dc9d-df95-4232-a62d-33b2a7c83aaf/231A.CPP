
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  // your code goes here
  int n;
  int count = 0;
  int problems = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int arr[3];
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      if (arr[i] == 1) {
        count++;
      }
    }
    if (count >= 2) {
      problems++;
      count = 0;
    } else {
      count = 0;
    }
  }
  cout << problems << endl;

  return 0;
}