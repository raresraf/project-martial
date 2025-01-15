#include <iostream>

using namespace std;

int main() {
  string word;
  cin >> word;

  auto count_0 = 0;
  auto count_1 = 0;

  for (auto c : word) {
    if (c == '0') {
      count_0++;
      count_1 = 0;
    } else {
      count_1++;
      count_0 = 0;
    }

    if (count_0 >= 7 || count_1 >= 7) {
      cout << "YES" << endl;
      return 0;
    }
  }

  cout << "NO" << endl;

  return 0;
}