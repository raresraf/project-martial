#include <iostream>
#include <string.h>

using namespace std;

main() {
  string word;
  int t;
  cin >> t;
  while (t > 0) {
    cin >> word;
    int n = word.length();
    if (word.length() <= 10)
      cout << word << endl;
    else
      cout << word[0] << n - 2 << word[n - 1] << endl;
    t = t - 1;
  }
}
