#include <iostream>
using namespace std;
string in, out;
int i, j, k, l, m, len;

void ifexist() {
  len = in.length();
  out = "NO";
  for (i = 0; i <= len - 1; i++) {
    if (in[i] == 'h') {
      for (j = i + 1; j <= len - 1; j++) {
        if (in[j] == 'e') {
          for (k = j + 1; k <= len - 1; k++) {
            if (in[k] == 'l') {
              for (l = k + 1; l <= len - 1; l++) {
                if (in[l] == 'l') {
                  for (m = l + 1; m <= len - 1; m++) {
                    if (in[m] == 'o')
                      out = "YES";
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

int main() {
  cin >> in;
  ifexist();
  cout << out << endl;
  return 0;
}