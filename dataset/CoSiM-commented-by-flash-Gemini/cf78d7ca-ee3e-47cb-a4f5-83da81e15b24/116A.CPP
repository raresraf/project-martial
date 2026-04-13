/**
 * @file 116A.CPP
 * @brief This program determines if two input strings can be made identical
 *        by swapping exactly two characters in one of the strings. This is
 *        checked by verifying if they have the same character composition
 *        and differ at exactly two positions.
 * Algorithm: String comparison, Difference counting, Sorting for character composition check.
 * Time Complexity: O(L log L) where L is the length of the input strings,
 *                  primarily due to the sorting operation. String traversal is O(L).
 * Space Complexity: O(L) for storing string copies during sorting.
 */
#include <bits/stdc++.h>
// typedef vector<int> vint;
#define pb push_back
using namespace std;
/*
// Functional Utility: This commented-out block provides an alternative implementation
//                     to check if two strings are "swappable" by directly identifying
//                     mismatched positions.
int main()
{
  string s,ss;
  cin>>s>>ss;
  vector<int> vec;
  int l1=s.length();
  int l2=ss.length();
  if(l1!=l2)cout<<"NO"<<endl;
  else{
       for(int i=0;i<l1;i++){
       if(s[i]!=ss[i])vec.pb(i);

   }

//for(int i=0;i<vec.size();i++)cout<<vec[i]<<" ";
   int l3=vec.size();
   if(l3==2)
   {
       if(s[vec[0]]==ss[vec[1]]  && s[vec[1]]==ss[vec[0]])
        cout<<"YES"<<endl;
       else cout<<"NO"<<endl;
   }
   else cout<<"NO"<<endl;
  }
  return 0;
}
*/
int main() {
  string s, ss; // Input strings.
  int count = 0; // Counter for the number of differing characters between s and ss.
  // Reads the two input strings.
  cin >> s >> ss;
  /**
   * Block Logic: Compares characters at corresponding positions in `s` and `ss`
   *              and increments `count` for each differing pair.
   * Pre-condition: `s` and `ss` are valid strings.
   * Invariant: `count` accurately reflects the number of positions where `s[i]` != `ss[i]`.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != ss[i]) // Checks if characters at current position are different.
      count++;         // Increments the difference counter.
  }
  // Sorts string `s` alphabetically to check for character composition.
  sort(s.begin(), s.end());
  // Sorts string `ss` alphabetically to check for character composition.
  sort(ss.begin(), ss.end());
  /**
   * Block Logic: Determines if the strings are "swappable" based on the number of
   *              differences and if they have the same character composition.
   * Invariant: Prints "YES" if `count` is exactly 2 and `s` and `ss` (after sorting) are identical,
   *            indicating a valid two-character swap scenario. Otherwise, prints "NO".
   */
  if (count == 2 && s == ss) // If exactly two characters differ and strings have same character multiset.
    cout << "YES" << endl;    // Strings can be made identical by swapping two characters.
  else
    cout << "NO" << endl; // Strings are not swappable or have different character compositions.
}
