#include <iostream>
#include <string.h>

using namespace std;

int main() {
  string s1, s2;
  int i;
  cin >> s1 >> s2;

  for (i = 0; i < s1.length(); i++) // passando tudo para minuscula
  {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  if (s1 == s2)
    cout << "0\n";
  if (s1 < s2)
    cout << "-1\n";
  if (s1 > s2)
    cout << "1\n";

  return 0;
}