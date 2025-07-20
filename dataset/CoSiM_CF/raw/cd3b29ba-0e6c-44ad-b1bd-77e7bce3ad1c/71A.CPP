#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  string *words = new string[n];
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }
  for (int i = 0; i < n; i++) {
    if (words[i].length() <= 10)
      cout << words[i] << endl;
    else {
      string firstL(1, words[i].at(0));
      string middleN = to_string((1, words[i].length() - 2));
      string lastL(1, words[i].at(words[i].length() - 1));
      cout << firstL + middleN + lastL << endl;
    }
  }
  delete[] words;
  return 0;
}
