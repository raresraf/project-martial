#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  char array[n];
  for (int i = 0; i < n; i++) {
    cin >> array[i];
  }
  int count = 0;
  int temp = 1;
  for (int i = 1; i < n; i++) {
    if (array[i] == array[i - temp]) {
      count++;
      temp++;
    } else
      temp = 1;
  }
  cout << count << endl;
  return 0;
}
