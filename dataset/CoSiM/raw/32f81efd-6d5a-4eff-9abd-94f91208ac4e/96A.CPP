#include <iostream>
#include <string>
using namespace std;
string in, out;
int i, len;

void ifsame() {
  len = in.length();
  for (i = 0; i <= len - 1; i++) {
    if (in[i] == in[i + 1] && in[i + 1] == in[i + 2] &&
        in[i + 2] == in[i + 3] && in[i + 3] == in[i + 4] &&
        in[i + 4] == in[i + 5] && in[i + 5] == in[i + 6]) {
      out = "YES";
      break;
    } else
      out = "NO";
  }
}

int main() {
  cin >> in;
  ifsame();
  cout << out << endl;
  return 0;
}