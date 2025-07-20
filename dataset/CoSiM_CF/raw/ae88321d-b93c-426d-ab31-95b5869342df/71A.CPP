#include <iostream>
using namespace std;
int main() {
  int n, l;
  string word;
  cin >> n;
  while (n--) {
    cin >> word;
    l = word.length();
    if (l > 10) {
      cout << word[0] << l - 2 << word[l - 1] << endl;
    } else {
      cout << word << endl;
    }
  }
}
