#include <iostream>
#define ll long long
#define ld long double

int main() {
  int n, k;
  std::cin >> n >> k;
  if (k > n) {
    std::cout << 0;
  }
  int tablica[n];
  for (int i = 0; i < n; i++) {
    tablica[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    std::cin >> tablica[i];
  }

  int output = 0;
  int wyznaczony;
  if (tablica[k - 1] > 0) {
    wyznaczony = tablica[k - 1];
  }

  for (int i = 0; i < n; i++) {
    if (tablica[i] > 0) {
      if (tablica[i] >= wyznaczony) {
        output++;
      }
    }
  }

  std::cout << output;
  return 0;
}