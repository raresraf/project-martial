#include <iostream>
#include <string>
using namespace std;
string in, out;
int i, len;

void makenewstring() {
  len = in.length();
  for (i = 0; i <= len - 1; i++) {
    if (in[i] != 'A' && in[i] != 'O' && in[i] != 'Y' && in[i] != 'E' &&
        in[i] != 'U' && in[i] != 'I' && in[i] != 'a' && in[i] != 'o' &&
        in[i] != 'y' && in[i] != 'e' && in[i] != 'u' && in[i] != 'i') {
      out += '.';
      out += towlower(in[i]);
    }
  }
}

int main() {
  cin >> in;
  makenewstring();
  cout << out << endl;
  return 0;
}
