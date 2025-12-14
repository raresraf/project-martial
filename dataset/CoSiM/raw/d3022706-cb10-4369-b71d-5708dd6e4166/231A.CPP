#include <iostream>
#include <string>
#define ll long long
#define ld long double

int main() {
  int n, output = 0;
  std::cin >> n;
  int a[n][3];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      std::cin >> a[i][j];
    }
  }

  for (int i = 0; i < n; i++) {
    if (a[i][0] == 1 && a[i][0] == a[i][1]) {
      output++;
      continue;
    } else if (a[i][1] == 1 && a[i][1] == a[i][2]) {
      output++;
      continue;
    } else if (a[i][0] == 1 && a[i][0] == a[i][2]) {
      output++;
      continue;
    }
  }

  std::cout << output;

  return 0;
}