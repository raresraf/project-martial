#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  cin >> n;

  string words[n];
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  cout << "\n";
  for (int i = 0; i < n; i++) {
    if (words[i].length() <= 10)
      cout << words[i] << "\n";
    else {
      string str = words[i];
      char c = str[0];
      char l = str[str.length() - 1];
      cout << c << str.length() - 2 << l << endl;
    }
  }
  return 0;
}
