#include <iostream>
#define ll long long
#define ld long double
#include <string>

int main() {
  int n;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    std::string word;
    std::cin >> word;
    if (word.length() > 10) {
      int dlugosc = word.length() - 2;
      std::cout << word[0];
      int ile = 0;
      for (int j = 0; j < dlugosc; j++) {
        ile++;
      }
      std::cout << ile << word[dlugosc + 1] << "\n";
    } else {
      std::cout << word << "\n";
    }
  }

  return 0;
}