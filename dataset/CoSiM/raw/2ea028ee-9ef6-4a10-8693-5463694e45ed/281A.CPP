
#include <bits/stdc++.h>

using namespace std;

string a;
char f;

int main() {
  cin >> f >> a;
  if ('a' <= f && f <= 'z')
    f -= 'a' - 'A';
  cout << f << a;
}
