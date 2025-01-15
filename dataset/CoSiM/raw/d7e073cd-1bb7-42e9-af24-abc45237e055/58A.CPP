#include <bits/stdc++.h>

using namespace std;

int main() {
  string a;
  cin >> a;
  for (int i = 0; i < a.length(); i++)
    for (int j = i + 1; j < a.length(); j++)
      for (int k = j + 1; k < a.length(); k++)
        for (int l = k + 1; l < a.length(); l++)
          for (int m = l + 1; m < a.length(); m++)
            if (a[i] == 'h' && a[j] == 'e' && a[k] == 'l' && a[l] == 'l' &&
                a[m] == 'o') {
              cout << "YES" << endl;
              return 0;
            }
  cout << "NO" << endl;
  return 0;
}
