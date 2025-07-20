#include <iostream>
#include <string>

using namespace std;

bool vowel(char a) {
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'y')
    return true;
  return false;
}

int main() {
  string read;
  cin >> read;
  string ans = "";
  for (int i = 0; i < read.size(); ++i) {
    ans += ".";
    char a = tolower(read[i]);
    if (vowel(a))
      ans.pop_back();
    else
      ans += a;
  }
  cout << ans << endl;
}