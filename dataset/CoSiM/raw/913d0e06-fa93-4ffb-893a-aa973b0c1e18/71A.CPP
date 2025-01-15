#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, i, l, j;
  string word;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> word;
    l = word.size();
    if (l <= 10)
      cout << word << endl;
    else
      cout << word[0] << l - 2 << word[l - 1] << endl;
  }
}