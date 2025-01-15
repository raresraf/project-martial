#include <iostream>
#include <string>
using namespace std;

int main() {
  string inp, out = "";
  cin >> inp;
  for (int i = 0; i < inp.size(); i++) {
    if (inp[i] == 'a' || inp[i] == 'A' || inp[i] == 'i' || inp[i] == 'I' ||
        inp[i] == 'e' || inp[i] == 'E' || inp[i] == 'o' || inp[i] == 'O' ||
        inp[i] == 'u' || inp[i] == 'U' || inp[i] == 'y' || inp[i] == 'Y') {
      continue;
    } else {
      out += ".";
      out += tolower(inp[i]);
    }
  }
  cout << out << endl;
}
