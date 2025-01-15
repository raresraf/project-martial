#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int conver(char a) { return a - 48; }

int main() {
  int n;
  cin >> n;
  char b[n];
  char prev;
  int total = 0;
  for (int i = 0; i < n; ++i) {
    cin >> b[i];
  }
  prev = b[0];

  for (int i = 1; i < n; ++i) {
    if (b[i] == prev) {
      total += 1;
    } else {
      prev = b[i]; // okay
    }
  }
  cout << total << endl;
}