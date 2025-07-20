//       In the name of         ****                      God ****

/*

Copyright (C)          JadedBeast

                                    Morocco

created : 9/2/2019

*/

#include <bits/stdc++.h>

using namespace std;

int main(void) {
  int n, t;
  string word;
  vector<int> occurence;
  cin >> n >> t;
  cin >> word;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (word[j] == 'B' && word[j + 1] == 'G') {
        occurence.push_back(j);
      }
    }
    for (int x : occurence) {
      swap(word[x], word[x + 1]);
    }
    occurence.clear();
  }

  cout << word << endl;
  return 0;
}
