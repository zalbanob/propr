# updateF <- function(pairwise, method="connectivity") {
#     WITH_NVTX("updateF", 0, {
# # genewiseResults(pairwise=pd@results, method=XX). 
# # The pairwise results have 4 important columns: 
# # - Partner,  Pair, Theta and FDR. 
# # - Partner and Pair are the gene index of the gene pair, 
# # - Theta is the differential metric and 
# # - FDR is the significance metric. 
# #
# # The idea is to take all the pairs in which a specific gene is involved (Gene set S with size = Ngenes - 1, as it is all vs all) and summarize the thetas somehow. 
# # There are 4 different “summary” methods:
# # Connectivity: 
# # - For each gene:
# #   - Sum the number of significant pairs (FDR lower than 0.05). 
# #   - Basically counting the number of significant pairs. 
# # Weighted connectivity: 
# # - For each gene:
# #   - Sum the (1 - theta) in the significant pairs (FDR lower than 0.05). 
# #   - Instead of adding +1 for each significant pair, as it was done in connectivity, 
# #       - you weight with theta: +1*(1-theta)
# # Mean: 
# # - For each gene:
# # - Take all pairs (not only the significant ones) and compute the average. 
# # Enrichment Score: 
# # - This is a bit more sophisticated. 
# #   - Basically we take all the pairs (or a randomly selected subset), 
# #       - we rank them with the lower thetas (more significant) first,
# #       - and we look at the position of the pairs for a given gene. 
# #       - If they are located rather at the beginning, 
# #           - we can say that the gene is enriched in significant pairs and therefore more likely to be differentially expressed. 
#     if (method == "connectivity") {

#     } else if (method == "weighted_connectivity") {

#     } else if (method == "mean") {

#     } else if (method == "enrichment_score") {

#     }else {
#         stop("Invalid method: ", method)
#     }
#     })
# }
