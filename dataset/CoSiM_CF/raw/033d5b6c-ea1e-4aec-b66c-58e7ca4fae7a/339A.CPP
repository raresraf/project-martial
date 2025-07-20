#include <iostream>
#include <string>
using namespace std;
int main() {
  string in, out;
  int c1 = 0, c2 = 0, c3 = 0;
  cin >> in;
  for (int i = 0; i < in.length(); i++) {
    if (in[i] == '1')
      c1++;
    else if (in[i] == '2')
      c2++;
    else if (in[i] == '3')
      c3++;
  }
  out = "";
  for (int i = 1; i <= c1; i++) {
    out = out + "1";
    out = out + "+";
  }
  for (int i = 1; i <= c2; i++) {
    out = out + "2";
    out = out + "+";
  }
  for (int i = 1; i <= c3; i++) {
    out = out + "3";
    out = out + "+";
  }
  for (int i = 0; i < out.length() - 1; i++) {
    cout << out[i];
  }
}
