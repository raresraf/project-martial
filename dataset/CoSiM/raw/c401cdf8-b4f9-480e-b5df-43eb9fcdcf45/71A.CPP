#include <iostream>
#include <string.h>
using namespace std;
#define MAXN 100

int main() {
  int N, l;
  cin >> N;

  for (int i = 0; i < N; i++) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      cout << s.at(0) << s.length() - 2 << s.at(s.length() - 1) << endl;
    } else {
      cout << s << endl;
    }
  }
}
