#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main(void) {
  string line;
  cin >> line;
  vector<char> ans;
  for (char c : line) {
    if (c >= 'A' && c <= 'Z')
      c = c - 'A' + 'a';
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {

    } else {
      ans.push_back(c);
    }
  }
  for (int i = 0; i < ans.size(); i++) {
    cout << '.' << ans[i];
  }
  cout << endl;

  return 0;
}