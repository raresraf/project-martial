/**
 * @file 116A.CPP
 * @brief Determines if two strings can be made identical with exactly one swap.
 * @note The filename `116A.CPP` is likely a misnomer, as the code's logic does not
 * correspond to the "Tram" problem (116A) but rather a common string manipulation problem.
 *
 * This file contains a solution to a problem that checks if a string 's' can be
 * transformed into another string 'ss' by swapping exactly one pair of characters in 's'.
 * The file includes two implementations within 'main' functions; the first is
 * commented out, and the second is the active one.
 *
 * @b Active Algorithm: Anagram and Difference Count
 *   This approach verifies two conditions:
 *   1. The strings must be anagrams of each other (checked by sorting both and comparing).
 *   2. There must be exactly two positions where the characters differ.
 * @b Time_Complexity: O(L*log(L)) due to the sorting of strings of length L.
 * @b Space_Complexity: O(L) or O(log(L)) depending on the space used by the sort implementation.
 *
 * @b Commented-out Algorithm: Direct Swap Check
 *   This approach identifies the indices where the strings differ and checks if there
 *   are exactly two such indices. It then verifies if the characters at these positions
 *   are mutually swapped between the two strings.
 *   Time Complexity: O(L). Space Complexity: O(L) in the worst case for the vector.
 */
#include <bits/stdc++.h>
// typedef vector<int> vint;
#define pb push_back
using namespace std;
/*
// This commented-out block contains an alternative O(L) solution.
int main()
{
  string s,ss;
  cin>>s>>ss;
  vector<int> vec;
  int l1=s.length();
  int l2=ss.length();
  // It first ensures the lengths are equal.
  if(l1!=l2)cout<<"NO"<<endl;
  else{
       // It finds all differing character positions.
       for(int i=0;i<l1;i++){
       if(s[i]!=ss[i])vec.pb(i);

   }

//for(int i=0;i<vec.size();i++)cout<<vec[i]<<" ";
   // It then checks if there are exactly two differing positions.
   int l3=vec.size();
   if(l3==2)
   {
       // Finally, it verifies that the characters at these positions are swapped.
       if(s[vec[0]]==ss[vec[1]]  && s[vec[1]]==ss[vec[0]])
        cout<<"YES"<<endl;
       else cout<<"NO"<<endl;
   }
   else cout<<"NO"<<endl;
  }
  return 0;
}*/
int main() {
  string s, ss;
  // 'count' tracks the number of positions with differing characters.
  int count = 0;
  cin >> s >> ss;
  /**
   * @brief Count the number of positions at which the strings s and ss differ.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != ss[i])
      count++;
  }

  // Sort both strings to check if they are anagrams.
  sort(s.begin(), s.end());
  sort(ss.begin(), ss.end());
  /**
   * @brief Check the two conditions for a single valid swap.
   * 1. The number of differing positions must be exactly 2.
   * 2. The strings must be anagrams (i.e., contain the same characters with the
   *    same frequencies), which is verified by comparing the sorted strings.
   */
  if (count == 2 && s == ss)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
