#include <iostream>
#include <string>
using namespace std;

int main() {
  string words;
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> words;
    if (words.length() > 10) {
      cout << words[0];
      cout << words.length() - 2;
      cout << words[words.length() - 1] << "\n";
    } else {
      cout << words << "\n";
    }
  }
  return 0;
}
