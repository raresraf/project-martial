#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int array[2 * n] = {0};
  for (int i = 0; i < n; i++) {
    int out;
    int in;
    cin >> out;
    cin >> in;
    if (i == 0)
      array[1] = in;
    else {
      array[2 * i] = array[2 * i - 1] - out;
      array[2 * i + 1] = array[2 * i] + in;
    }
  }
  int maximum = *max_element(array, array + 2 * n);
  cout << maximum << endl;
  return 0;
}
