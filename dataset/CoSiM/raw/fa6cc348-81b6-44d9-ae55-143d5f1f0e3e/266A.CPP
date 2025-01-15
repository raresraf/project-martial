#include <iostream>
#include <new>;
using namespace std;

int main() {
  int n;
  cin >> n;
  char *s = new char[n + 1];
  int i;
  for (i = 0; i < n; i++)
    cin >> s[i];
  int c = 0;
  for (i = 0; i < n - 1; i++)
    if (s[i] == s[i + 1])
      c++;

  cout << c << endl;

  delete s;
  s = NULL;
  return 0;
}
