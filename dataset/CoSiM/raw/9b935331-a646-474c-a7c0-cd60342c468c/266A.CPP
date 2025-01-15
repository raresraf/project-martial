#include <iostream>

using namespace std;

int main() {
  string word;
  int test;

  cin >> test >> word;

  auto count = 0;

  for (auto i = 0; i < test - 1; i++) {
    if (word[i] == word[i + 1])
      count++;
  }

  cout << count << endl;

  return 0;
}
