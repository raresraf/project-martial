#include <iostream>
using namespace std;
main() {
  string word;
  cin >> word;
  int x = 1;
  int n = word.length();
  for (int i = 1; i < n; i++) {
    if (word[i] == word[i - 1]) {
      x = x + 1;
      // cout<<x<<endl;
      if (x >= 7) {
        cout << "YES";
        return 0;
      }
    } else
      x = 1;
  }
  cout << "NO";
}
