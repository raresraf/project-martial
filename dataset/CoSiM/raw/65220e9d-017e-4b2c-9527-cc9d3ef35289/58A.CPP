#include <iostream>
using namespace std;
main() {
  string word;
  cin >> word;
  string match = "hello";
  int k = 0;
  int n = word.length();
  for (int i = 0; i < n; i++) {
    if (word[i] == match[k])
      k++;
  }
  if (k == match.length())
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
