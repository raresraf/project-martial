//
// Created by Taewoo Kim on 3/23/2017.
//

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  getchar(); // consume newline
  int r = 0, g = 0, b = 0, repeat = 0;
  char prev = '\0';

  for (int i = 0; i < n; i++) {
    char c = getchar();
    if (c == prev)
      repeat++;
    prev = c;
  }
  cout << repeat;
  return 0;
}