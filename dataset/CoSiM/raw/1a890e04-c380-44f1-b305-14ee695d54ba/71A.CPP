#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, i, l;
  string word;
  cin >> n;
  string wordArray[n];
  for (i = 0; i < n; i++)
    cin >> wordArray[i];

  for (i = 0; i < n; i++) {
    word = wordArray[i];
    l = word.length();
    if (l < 11)
      cout << word << endl;
    else {
      l -= 2;
      cout << word.at(0) << l << word.at(l + 1) << endl;
    }
  }

  return 0;
}